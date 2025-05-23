(define (problem tmp)
	(:domain workspace)
	(:objects
		rack - receptacle
		icecream - box
		salt - box
	)
	(:init
		(aligned rack)
		(poslimit rack)
		(inworkspace rack)
		(on rack table)
		(on icecream table)
		(inhand salt)
	)
	(:goal (and))
)
