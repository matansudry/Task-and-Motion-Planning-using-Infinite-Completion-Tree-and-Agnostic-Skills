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
		(beyondworkspace rack)
		(on rack table)
		(on salt rack)
		(on icecream table)
	)
	(:goal (and))
)
