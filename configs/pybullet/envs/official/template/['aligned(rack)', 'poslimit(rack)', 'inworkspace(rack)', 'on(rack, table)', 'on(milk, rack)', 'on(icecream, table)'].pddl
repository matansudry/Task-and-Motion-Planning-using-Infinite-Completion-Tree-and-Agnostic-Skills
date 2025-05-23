(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		milk - box
		icecream - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on milk rack)
		(on icecream table)
	)
	(:goal (and))
)
